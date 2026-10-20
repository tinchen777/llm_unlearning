#!/bin/bash
# =============================================================================
# TOFU unlearning baselines (OpenUnlearning trainers) for ONE (model, forget set); the methods are listed in _common.sh.
# Entry: src/train.py (mode=unlearn); experiment=unlearn/tofu/default (forget + retain data, balanced on the forget set)
#        or unlearn/tofu/idk (IdkDPO). The TOFU metrics are evaluated at the start and after every epoch by the trainer
#        (eval.tofu in the experiment), forget quality / privleak against the official retain model (RETAIN_LOGS).
#
# Every method is trained from the official TOFU-full model:  GradAscent: -forget loss;  GradDiff: forget ascent + retain NLL;
# NPO / SimNPO: bounded negative preference;  IdkDPO: prefer "I don't know";  RMU: random steering vector on one layer.
#
# Output: saves/unlearn/tofu_<MODEL>_<FORGET_SPLIT>/<METHOD>/   (see task_name() in _common.sh)
#   checkpoint-<step>/evals/TOFU_SUMMARY.json   TOFU metrics of every epoch (step 0 = the original model)
# Loop over models / forget sets: scripts/unlearn/run.sh.   MUSE: scripts/unlearn/3_unlearn.sh.
# =============================================================================
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for METHOD in "${METHODS[@]}"; do
  echo start ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
  python src/train.py --config-name=unlearn \
    $(method_args ${METHOD}) \
    model=${FULL_MODEL} \
    "${SPLIT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    task_name=$(task_name ${METHOD})
    # --cfg job --resolve
  echo end ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
done
