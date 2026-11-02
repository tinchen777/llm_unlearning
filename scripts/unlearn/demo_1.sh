#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# source "${SCRIPT_DIR}/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

method_args() {
  case "$1" in
    IdkDPO)
      # DPO with "I don't know" answers
      echo "experiment=unlearn/tofu/idk trainer=DPO trainer.args.eval_on_start=False"
      ;;

    PDU)
      # retain_loss_eps has no default
      echo "experiment=unlearn/tofu/default trainer=PDU trainer.method_args.retain_loss_eps=0.3"
      ;;

    GradAscent|GradDiff|NPO|SimNPO|RMU|CEU|WGA|UNDIAL|SatImp|BoundedGradDiff)
      echo "experiment=unlearn/tofu/default trainer=$1"
      ;;

    *)
      echo "unknown method '$1'" >&2
      return 1
      ;;
  esac
}

METHODS=(
    GradAscent
    # GradDiff
    # NPO
    # SimNPO
    # IdkDPO
    # RMU
)
MODEL=Llama-3.2-1B-Instruct
FULL_MODEL=tofu/tofu_${MODEL}_full

FORGET_SPLIT=forget01
HOLDOUT_SPLIT=holdout01
RETAIN_SPLIT=retain99

SPLIT_ARGS=(
  "forget_split=${FORGET_SPLIT}"
  "retain_split=${RETAIN_SPLIT}"
  "holdout_split=${HOLDOUT_SPLIT}"
)
RETAIN_LOGS=saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for METHOD in "${METHODS[@]}"; do
  echo start ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
  python src/train.py --config-name=unlearn \
    $(method_args ${METHOD}) \
    model=${FULL_MODEL} \
    "${SPLIT_ARGS[@]}" \
    retain_logs_path=${RETAIN_LOGS} \
    "${EXTRA_ARGS[@]}" \
    task_name=ttestt/${METHOD}_1 \
    # --cfg job --resolve
  echo end ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
done
