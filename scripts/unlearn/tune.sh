#!/bin/bash
# Tuning of the baselines (1_unlearn.sh): the learning rate (and the weights dtype) of every method on ONE (model, forget set). forget01 has 40 samples,
# i.e. 2 optimizer steps per epoch (batch 8 x accumulation 4); the protocol is 5 epochs = 10 steps (EPOCHS, _common.sh), so the learning rate has to be larger than the default 1e-5.
# The trainer evaluates after every epoch, so the epoch is picked afterwards (scripts/analysis/baseline_table.py), not swept here.
#
#   bash scripts/unlearn/tune.sh [GPU_ID]                                # LRS="5e-5 1e-4", the 6 methods, forget01 of the 1B model
#   LRS="2e-5 5e-5 1e-4 2e-4" METHODS="NPO GradDiff" bash scripts/unlearn/tune.sh 0
#   DTYPES="bf16 float32" bash scripts/unlearn/tune.sh 0                 # also fp32 weights (bf16 = the default of the model config)
#   MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 bash scripts/unlearn/tune.sh 0
#   DRY_RUN=1 bash scripts/unlearn/tune.sh                               # only print the runs
#
# PROBE=1 + EVAL_PROFILE=fast (the defaults here): the metrics of configs/eval/tofu_fast.yaml + the NEAR probe + detectability after every epoch, so every run
# gives the points of scripts/analysis/pareto.py with ud_* at once. PROBE=0 / EVAL_PROFILE=full for the TOFU-only / the complete metric set (forget_quality, which
# baseline_table.py needs to pick an epoch).
# Output: saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET_SPLIT>_lr<LR>_ep<EPOCHS>[_fp32]/ ; the table with the best epoch of every run:
#   python scripts/analysis/baseline_table.py saves/tofu_unlearn
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/../.." || exit 1

GPU_ID=${1:-0}
read -r -a LRS <<< "${LRS:-5e-5 1e-4}"
read -r -a DTYPES <<< "${DTYPES:-bf16}"
export PROBE=${PROBE:-1}
export EVAL_PROFILE=${EVAL_PROFILE:-fast}

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
